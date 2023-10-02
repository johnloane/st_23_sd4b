import numpy as np

def main():
    #add_two_vectors_using_lists()
    #add_two_vectors_using_numpy()
    #test_multi_dim_arrays()
    #index_into_multi_array()
    #one_dim_slicing()
    #reshaping_array()
    #test_conditionals()
    #array_flattening()
    #test_transpose()
    #matrix_multiplication()
    #test_stacking()
    test_depth_stacking()


def test_depth_stacking():
    x = np.arange(4).reshape(2, 2)
    y = x ** 2
    z = np.dstack((x, y))
    print(z)
    print(z.shape)


def test_stacking():
    x = np.arange(4).reshape(2, 2)
    y = np.arange(4, 8).reshape(2, 2)
    z = np.hstack((x, y))
    w = np.column_stack((x, y))
    v = np.concatenate((x, y), axis=1)

    h = np.vstack((x, y))
    i = np.row_stack((x, y))
    j = np.concatenate((x, y), axis=0)
    print(h)
    
    print(z==w)
    print(z==v)


def matrix_multiplication():
    mat_a = np.matrix([0, 3, 5, 5, 5, 2]).reshape(2, 3)
    mat_b = np.matrix([3, 4, 3, -2, 4, -2]).reshape(3, 2)
    print(mat_a)
    print(mat_a * mat_b)


def test_transpose():
    x = np.arange(9).reshape(3, 3)
    print(x)
    print(x.transpose())
    print(x.T)
    print(np.zeros((3, 2)))
    print(np.ones((3, 2)))
    print(np.eye(3))
    print(np.random.rand(4,4))


def array_flattening():
    x = np.arange(9).reshape(3, 3)
    print(x)
    ravelled_array = x.ravel()
    ravelled_array[0] = 1000000
    print(x, ravelled_array)


def test_conditionals():
    x = np.arange(10)
    print(x[x > 5])
    print(x.max())
    print(x.min())


def reshaping_array():
    x = np.arange(18).reshape(3, 3, 2)
    print(x[1])

def one_dim_slicing():
    x = np.arange(1, 10)
    print(x[:7])



def index_into_multi_array():
    x = np.arange(3)
    y = np.arange(3, 6)
    z = np.arange(255, 258)
    multi_dim_array = np.array([x, y, z], dtype=np.int8)
    print(multi_dim_array)
    print(multi_dim_array.dtype)


def test_multi_dim_arrays():
    x = np.arange(3)
    y = np.arange(3)
    z = np.arange(3)
    multi_dim_array = np.array([x, y, z])
    print(multi_dim_array)
    print(multi_dim_array.shape)
    w = np.linspace(1, 10, 50)
    print(w)
    b = np.arange(1, 30 , 3)
    bl = np.linspace(1, 30, 3)
    print(bl)
    bl = np.linspace(1, 30, 3, endpoint=False)
    print(bl)


def add_two_vectors_using_numpy():
    array_two = np.arange(1, 4) ** 2
    array_three = np.arange(1, 4) ** 3
    array_sum = array_two + array_three
    print(array_sum)
    sample_array = np.array([1, 2, 3, 4, 5])
    print(np.power(sample_array, 4))
    print(np.negative(sample_array))
    print(np.exp(sample_array))
    print(np.log(sample_array))
    print(np.sin(sample_array))



def add_two_vectors_using_lists():
    list_two = list(range(1, 4))
    list_three = list(range(1, 4))
    list_sum = []

    for i in range(3):
        list_two[i] = list_two[i] ** 2
        list_three[i] = list_three[i] ** 3
        list_sum.append(list_two[i] + list_three[i])
    print(list_sum)

main()