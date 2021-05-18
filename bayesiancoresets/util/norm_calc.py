import numpy as np
import cupy as cp

class NormCalc():
    @classmethod
    def memory_avail(cls):
        dev = cp.cuda.Device(0)
        return dev.mem_info[0]
    
    def __init__(self, file_list, num_rows, num_cols, gpu_list):
        self.files = file_list
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.gpu_list = gpu_list
        self.num_gpus = len(gpu_list)

    def dot_product(self, coef_vector):
        loc = 0
        product = np.zeros(self.num_rows)
        for i in range(len(self.files)):
            with cp.cuda.Device(self.gpu_list[i % self.num_gpus]):
                coef = cp.asarray(coef_vector)
                matrix_part = cp.load(self.files[i])
                n = matrix_part.shape[0]
                prod = cp.matmul(matrix_part, coef)
                product[loc:loc+n] = prod.get()
                loc = loc + n
        return product

    def multiply(self, factor, new_files):
        loc = 0
        norm = np.zeros(self.num_rows)
        for i in range(len(self.files)):
            with cp.cuda.Device(self.gpu_list[i % self.num_gpus]):
                coef = cp.asarray(factor)
                matrix_part = cp.load(self.files[i])
                n = matrix_part.shape[0]
                matrix_part = matrix_part * coef
                cp.save(new_files[i], matrix_part)
                norm_part = cp.sum(matrix_part, axis=1)
                norm[loc:loc+n] = norm_part.get()
                loc = loc + n
        return norm

    def get_norm(self, weights):
        loc = 0
        norm = np.zeros(self.num_cols)
        for i in range(len(self.files)):
            with cp.cuda.Device(self.gpu_list[i % self.num_gpus]):
                coef = cp.asarray(weights)
                matrix_part = cp.load(self.files[i])
                n = matrix_part.shape[1]
                matrix_part = matrix_part * coef
                norm_part = cp.sum(matrix_part**2, axis=0)
                norm = norm + norm_part.get()
        return norm
        
    def dot_product_normalized(self, coef_vector, norms):
        loc = 0
        product = np.zeros(self.num_rows)
        for i in range(len(self.files)):
            with cp.cuda.Device(self.gpu_list[i % self.num_gpus]): 
                norm_gpu = cp.asarray(norms)
                coef = cp.asarray(coef_vector)
                matrix_part = cp.load(self.files[i])
                n = matrix_part.shape[0]
                matrix_part = matrix_part / norm_gpu
                prod = cp.matmul(matrix_part, coef)
                product[loc:loc+n] = prod.get()
                loc = loc + n
        return product

    def dot_product_normalized_trans(self, coef_vector, norms):
        product = np.zeros(self.num_cols)
        loc = 0
        for i in range(len(self.files)):
            with cp.cuda.Device(self.gpu_list[i % self.num_gpus]):
                norm_gpu = cp.asarray(norms)
                coef = cp.asarray(coef_vector)
                matrix_part = cp.load(self.files[i])
                matrix_part = matrix_part / norm_gpu
                matrix_part = cp.swapaxes(matrix_part, 0, 1)
                n = matrix_part.shape[1]
                prod = cp.matmul(matrix_part, coef[loc:loc+n])
                loc = loc + n
                product = product + prod.get()
        return product

    def get_col(self, col_idc):
        if col_idc>self.num_cols:
            raise ValueError("There are only " + self.num_cols + " columns in the array")
        col = np.zeros(self.num_rows)
        loc = 0
        for i in range(len(self.files)):
            matrix_part = np.load(self.files[i])
            n = matrix_part.shape[0]
            col[loc:loc+n] = matrix_part[:, col_idc]
            loc = loc + n
        return col
