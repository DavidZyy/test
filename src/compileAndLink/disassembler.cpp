#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <iostream>

int main(int argc, char **argv) {
  // Initialize LLVM targets and assembly printers
  LLVMInitializeAllTargetInfos();
  LLVMInitializeAllTargetMCs();
  LLVMInitializeAllDisassemblers();
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  
  // Load the binary file to be disassembled
  std::ifstream binary_file("my_binary_file", std::ios::binary);
  std::vector<uint8_t> binary_data(std::istreambuf_iterator<char>(binary_file), {});

  // Set up the target architecture and features
  std::string arch_name = "x86_64";
  std::string cpu_name = "generic";
  std::string features_str = "";
  auto target_triple = llvm::Triple(llvm::Triple::normalize(arch_name + "-unknown-linux-gnu"));
  auto target = llvm::TargetRegistry::lookupTarget(arch_name, target_triple, error_str);
  auto cpu = cpu_name.empty() ? "" : cpu_name;
  auto features = features_str.empty() ? "" : features_str;
  auto subtarget = target->createMCSubtargetInfo(target_triple.getTriple(), cpu, features);

  // Create the disassembler
  auto disasm = target->createMCDisassembler(*subtarget, context);

  // Set up the instruction and register info
  auto instr_info = target->createMCInstrInfo();
  auto reg_info = target->createMCRegInfo(target_triple.getTriple());

  // Disassemble the binary file
  uint64_t address = 0;
  uint64_t size = binary_data.size();
  llvm::MCInst instruction;
  llvm::MCDisassembler::DecodeStatus status;
  while (address < size) {
    status = disasm->getInstruction(instruction, size, binary_data, address, llvm::nulls(), llvm::nulls());
    if (status != llvm::MCDisassembler::Success) {
      std::cerr << "Failed to disassemble instruction at address " << address << std::endl;
      break;
    }
    std::cout << "0x" << std::hex << address << ":\t" << instr_info->getName(instruction.getOpcode()).str() << std::endl;
    address += instruction_size;
  }
  
  return 0;
}