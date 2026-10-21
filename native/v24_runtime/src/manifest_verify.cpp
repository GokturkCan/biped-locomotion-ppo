#include "biped_native/manifest_verify.h"

#include <openssl/evp.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace biped {

std::string Sha256HexOfFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("manifest_verify: cannot open " + path);

  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (!ctx) throw std::runtime_error("manifest_verify: EVP_MD_CTX_new failed");
  if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("manifest_verify: EVP_DigestInit_ex failed");
  }

  std::vector<char> buf(1 << 16);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = f.gcount();
    if (n > 0 && EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(n)) != 1) {
      EVP_MD_CTX_free(ctx);
      throw std::runtime_error("manifest_verify: EVP_DigestUpdate failed for " + path);
    }
  }

  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int digest_len = 0;
  if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("manifest_verify: EVP_DigestFinal_ex failed for " + path);
  }
  EVP_MD_CTX_free(ctx);

  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(digest_len * 2);
  for (unsigned int i = 0; i < digest_len; ++i) {
    out.push_back(kHex[(digest[i] >> 4) & 0xF]);
    out.push_back(kHex[digest[i] & 0xF]);
  }
  return out;
}

std::string ReadManifestArtifactSha256(const std::string& manifest_path,
                                        const std::string& artifact_filename) {
  std::ifstream f(manifest_path, std::ios::binary);
  if (!f) throw std::runtime_error("manifest_verify: cannot open " + manifest_path);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  const std::string key = "\"" + artifact_filename + "\"";
  const size_t key_pos = text.find(key);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("manifest_verify: artifact key " + key + " not found in " +
                              manifest_path);
  }
  const size_t sha_key_pos = text.find("\"sha256\"", key_pos);
  if (sha_key_pos == std::string::npos) {
    throw std::runtime_error("manifest_verify: no \"sha256\" field after " + key + " in " +
                              manifest_path);
  }
  // Find the opening quote of the value, after the colon following "sha256".
  const size_t colon_pos = text.find(':', sha_key_pos);
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("manifest_verify: malformed sha256 field in " + manifest_path);
  }
  const size_t value_start = text.find('"', colon_pos);
  if (value_start == std::string::npos) {
    throw std::runtime_error("manifest_verify: malformed sha256 value in " + manifest_path);
  }
  const size_t value_end = text.find('"', value_start + 1);
  if (value_end == std::string::npos) {
    throw std::runtime_error("manifest_verify: unterminated sha256 value in " + manifest_path);
  }
  return text.substr(value_start + 1, value_end - value_start - 1);
}

void VerifyArtifactHashOrThrow(const std::string& manifest_path, const std::string& artifact_path,
                                const std::string& artifact_filename) {
  const std::string expected = ReadManifestArtifactSha256(manifest_path, artifact_filename);
  const std::string actual = Sha256HexOfFile(artifact_path);
  if (expected != actual) {
    throw std::runtime_error("manifest_verify: HASH MISMATCH for " + artifact_filename +
                              " -- manifest says " + expected + ", file on disk hashes to " +
                              actual + " (" + artifact_path + "). Refusing to run with an "
                              "unverified artifact.");
  }
  std::printf("[manifest_verify] %s OK (sha256=%s)\n", artifact_filename.c_str(), actual.c_str());
}

}  // namespace biped
