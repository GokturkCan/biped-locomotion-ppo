// Tests for manifest_verify.cpp: sha256 of a file, manifest lookup and the
// refuse-on-mismatch contract. Uses temporary files only.
//
// Usage: test_startup_verification
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "biped_native/manifest_verify.h"
#include "check.h"

namespace fs = std::filesystem;

int main() {
  using testing::Check;
  const fs::path dir = fs::temp_directory_path() / "biped_native_manifest_test";
  fs::create_directories(dir);
  const fs::path artifact = dir / "artifact.bin";
  { std::ofstream(artifact, std::ios::binary) << "abc"; }

  // sha256("abc")
  const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  Check(biped::Sha256HexOfFile(artifact.string()) == abc, "sha256 of known content");

  const fs::path manifest = dir / "EXPORT_MANIFEST.json";
  { std::ofstream(manifest) << "{\"artifacts\": {\"artifact.bin\": {\"sha256\": \"" << abc
                            << "\", \"bytes\": 3}}}"; }
  Check(biped::ReadManifestArtifactSha256(manifest.string(), "artifact.bin") == abc,
        "manifest lookup");
  biped::VerifyArtifactHashOrThrow(manifest.string(), artifact.string(), "artifact.bin");
  Check(true, "matching hash accepted");

  { std::ofstream(artifact, std::ios::binary) << "abd"; }
  bool threw = false;
  try {
    biped::VerifyArtifactHashOrThrow(manifest.string(), artifact.string(), "artifact.bin");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "tampered artifact is refused");

  threw = false;
  try {
    biped::ReadManifestArtifactSha256(manifest.string(), "other.bin");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "unknown artifact key reported");

  threw = false;
  try {
    biped::Sha256HexOfFile((dir / "missing").string());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "missing file reported");

  fs::remove_all(dir);
  return testing::Finish("test_startup_verification");
}
