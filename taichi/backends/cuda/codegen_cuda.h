// The CUDA backend

#pragma once

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCUDA : public KernelCodeGen {
 public:
  CodeGenCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  FunctionType codegen() override;
};

TLANG_NAMESPACE_END
