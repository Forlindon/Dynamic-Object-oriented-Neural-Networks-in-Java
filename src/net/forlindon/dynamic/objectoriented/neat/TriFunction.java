package net.forlindon.dynamic.objectoriented.neat;

@FunctionalInterface
public interface TriFunction<T, U, V, R> {

    R apply(T t, U u, V v);

}
