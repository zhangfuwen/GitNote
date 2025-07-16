
## 如何判断一个点在三角形内部？

a,b,c 三点clock wise组件三角形。点p一定在a->b向量的右侧，b->c向量的右侧，c->a向量的右侧。
这也意味着
a->b 与a->p两个向量的差乘一定是指定另一面的。
b->c 与b->p同理。
c->a 与c->p同理。

```c++
constexpr vec minus(point a, point b) {
    return {b.x - a.x, b.y -a.y, b.z -a.x};
}
constexpr bool same_direction(vector a, vector b) {
    return  dot_product(a,b) > 0; // should be 1, but floating point computation...
}

constexpr bool in_triangle(point a, point b, point c, point p) {
    vec ab_vec = minus(a, b);
    vec ap_vec = minus(a, p);
    vec cross_1 = cross_product(ab_vec, ap_vec);
    
    
    vec bc_vec = minus(b, c);
    vec bp_vec = minus(b, p);
    vec cross_2 = cross_product(bc_vec, bp_vec);
    
    vec ca_vec = minus(c, a);
    vec cp_vec = minus(c, p);
    vec cross_p = cross_product(ca_vec, cp_vec);
    
    return same_direction(cross_1, cross_2) && same_direction(cross_2, cross_3);
}

```
