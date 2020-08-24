class A(object):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class B(A):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class C(A):

    def __init__(self, *args):
        print(args)
        super().__init__(*args)

c = C(3,4,5)
print(c)
b = B(1,2,3)
print(b)