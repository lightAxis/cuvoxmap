#pragma once

namespace cuvoxmap
{
    /**
     * @brief Check if GPU support is available.
     * This function cannot be used in a constexpr context.
     * @return true if GPU support is available, false otherwise.
     * @note This function is optional and can be used to check if GPU support is available.
     */
    bool GPU_SUPPORT();

    /**
     * @brief Link error if GPU is not supported.
     * This is optional and can be used to ensure that the code is only compiled if GPU support is available.
     */
    void LINK_ERROR_IF_NO_GPU();
}