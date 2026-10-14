"""Scoped, source-pinned substitution of the VBx training-embedding normalization expression."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading


VBX_CALL_SHA256 = "8baf21775d7add4f8b666ab480470c2748592074541f1cfe32a400ed118540d7"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_normalization(original, normalize):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != VBX_CALL_SHA256 or original.__closure__:
        raise ValueError("embedding_vbx_source_not_supported")
    tree = ast.parse(source)
    expected = ast.parse("train_embeddings_normed = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    if len(matches) != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("embedding_vbx_normalization_contract_changed")
    matches[0].value = ast.copy_location(ast.parse("__wx_native_embedding_normalize(train_embeddings)").body[0].value, matches[0].value)
    namespace = dict(original.__globals__)
    namespace["__wx_native_embedding_normalize"] = normalize
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native VBx normalization>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("embedding_vbx_signature_changed")
    return candidate


@contextlib.contextmanager
def native_embedding_normalization(context, records):
    global _ACTIVE
    from pyannote.audio.pipelines.clustering import VBxClustering

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("embedding_normalization_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4":
            raise ValueError("embedding_pyannote_version_not_supported")
        original = VBxClustering.__call__
        owner_thread = threading.get_ident()

        def normalize(embeddings):
            output, record = context.normalize(embeddings)
            record = dict(record)
            record["adapter_scope"] = "vbx_training_embeddings_before_centroid_linkage"
            record["original_call_sha256"] = VBX_CALL_SHA256
            record["pyannote_audio_version"] = "4.0.4"
            records.append(record)
            return output

        candidate = compile_native_normalization(original, normalize)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            return candidate(*args, **kwargs)

        _ACTIVE = True
        VBxClustering.__call__ = wrapped
        try:
            yield
        finally:
            VBxClustering.__call__ = original
            _ACTIVE = False
