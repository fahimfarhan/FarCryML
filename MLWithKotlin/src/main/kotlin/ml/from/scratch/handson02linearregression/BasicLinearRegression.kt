package ml.from.scratch.handson02linearregression

import org.jetbrains.kotlinx.multik.api.mk as np
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray


class BasicLinearRegression(
    private var M: Int = 0,
    private var N: Int = 0
) {

    /**
     * @brief:
     * Y_nx1 = X_nxm * W_mx1 + B_nx1
     * */

    private var bias: Float = 0f
    private var W_mx1: Any? = null // NDArray<Int, D1> = NDArray<Int, D1>()



    init {

    }

    private fun trialAndError() {
        val a = np.ndarray(np[1, 2, 3])
        val at: NDArray<Int, D1> = a.transpose()
    }


}