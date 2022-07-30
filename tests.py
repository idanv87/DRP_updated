

d={'a':[1,2],'b':[3,4]}
d2=dict()
[d2.setdefault(key ,d[key].copy()) for key in list(d) ]
d['a'].append(3)
print(d)

class X:
    def __int__(self):
        pass
    @classmethod
    def method(cls):
        print('h')
x=X()
X.method()