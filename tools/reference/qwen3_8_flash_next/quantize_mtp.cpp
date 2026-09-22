// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_nvfp4_expert_bank.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ninfer_quantize_mtp <input.ninfer> <out_gate_up.bin> <out_down.bin>\n";
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string out_gate_up = argv[2];
    const std::string out_down    = argv[3];

    try {
        std::cout << "Opening artifact: " << input_path << "\n";
        artifact::Reader reader(input_path);

        const auto* gate_up_desc = reader.find("mtp/layer/mlp/experts/gate_up");
        const auto* down_desc    = reader.find("mtp/layer/mlp/experts/down");
        if (!gate_up_desc || !down_desc) {
            std::cerr << "Error: MTP expert tensors not found in artifact!\n";
            return 1;
        }

        const auto gate_up_payload = reader.payload(*gate_up_desc);
        const auto down_payload    = reader.payload(*down_desc);

        std::cout << "gate_up source bytes: " << gate_up_payload.data.size() << "\n";
        std::cout << "down source bytes: " << down_payload.data.size() << "\n";

        const std::size_t gate_up_nvfp4_bytes =
            flash_next_nvfp4_expert_bank_payload_bytes(512, 1280, 2560);
        const std::size_t down_nvfp4_bytes =
            flash_next_nvfp4_expert_bank_payload_bytes(512, 2560, 640);

        std::cout << "Allocating device buffers: gate_up=" << gate_up_nvfp4_bytes
                  << " B, down=" << down_nvfp4_bytes << " B\n";
        DeviceBuffer d_gate_up(gate_up_nvfp4_bytes);
        DeviceBuffer d_down(down_nvfp4_bytes);

        const auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Quantizing gate_up to NVFP4...\n" << std::flush;
        quantize_bf16_expert_bank_to_nvfp4(gate_up_payload.data.data(), d_gate_up.p, 512, 1280, 2560,
                                           nullptr);

        std::cout << "Quantizing down to NVFP4...\n" << std::flush;
        quantize_bf16_expert_bank_to_nvfp4(down_payload.data.data(), d_down.p, 512, 2560, 640,
                                           nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "GPU quantization completed in " << elapsed_ms << " ms\n";

        std::cout << "Copying gate_up to host and saving to " << out_gate_up << "...\n";
        std::vector<std::byte> h_gate_up(gate_up_nvfp4_bytes);
        CUDA_CHECK(cudaMemcpy(h_gate_up.data(), d_gate_up.p, gate_up_nvfp4_bytes,
                              cudaMemcpyDeviceToHost));
        std::ofstream f_gate_up(out_gate_up, std::ios::binary);
        f_gate_up.write(reinterpret_cast<const char*>(h_gate_up.data()),
                        static_cast<std::streamsize>(h_gate_up.size()));
        f_gate_up.close();

        std::cout << "Copying down to host and saving to " << out_down << "...\n";
        std::vector<std::byte> h_down(down_nvfp4_bytes);
        CUDA_CHECK(cudaMemcpy(h_down.data(), d_down.p, down_nvfp4_bytes,
                              cudaMemcpyDeviceToHost));
        std::ofstream f_down(out_down, std::ios::binary);
        f_down.write(reinterpret_cast<const char*>(h_down.data()),
                        static_cast<std::streamsize>(h_down.size()));
        f_down.close();

        std::cout << "DONE: Successfully quantized and saved MTP expert tensors!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
