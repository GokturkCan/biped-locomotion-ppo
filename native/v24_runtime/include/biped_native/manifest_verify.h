// Verifies deployed artifact files (e.g. the ONNX policy) against the sha256
// hashes recorded in an export manifest, so the runtime refuses to run an
// unverified artifact. Deliberately minimal: it reads the scalar
//   "<artifact filename>": { "sha256": "<hex>", ... }
// shape that tools/export_onnx.py writes, using targeted text extraction
// instead of a full JSON parser.
#pragma once

#include <string>

namespace biped {

// Computes the lowercase hex SHA-256 digest of the file at `path` (OpenSSL
// EVP). Throws std::runtime_error if the file cannot be read.
std::string Sha256HexOfFile(const std::string& path);

// Extracts manifest["artifacts"][artifact_filename]["sha256"] from the JSON
// text at `manifest_path` via targeted string search (not a general JSON
// parser). Throws std::runtime_error if the file or the key cannot be found.
std::string ReadManifestArtifactSha256(const std::string& manifest_path,
                                        const std::string& artifact_filename);

// Convenience: throws std::runtime_error with a descriptive message unless
// Sha256HexOfFile(artifact_path) == ReadManifestArtifactSha256(manifest_path,
// artifact_filename). Call once per artifact (onnx, mjb) at startup, before
// settling/trusting the model.
void VerifyArtifactHashOrThrow(const std::string& manifest_path, const std::string& artifact_path,
                                const std::string& artifact_filename);

}  // namespace biped
