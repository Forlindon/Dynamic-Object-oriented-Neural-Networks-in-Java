package knot;

import connection.Connection;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public abstract class Knot {

    private final List<Connection> OUTBOUND;
    private final int LAYER_ID;
    private double value = 0.0;
    private double bias = Math.random() * 0.5;

    public Knot(int id) {
        this.LAYER_ID = id;
        this.OUTBOUND = new ArrayList<>();
    }

    public void connect(Knot other, Function<Knot, Connection> factory) {
        if (other.LAYER_ID <= this.LAYER_ID) throw new IllegalArgumentException("Invalid LAYER_ID");
        Connection c = factory.apply(other);
        if (this.OUTBOUND.contains(c)) throw new IllegalArgumentException("No duplicate connections");
        this.OUTBOUND.add(c);
    }

    public abstract double activation(double d);

    public void pop() {
        double d = activation(this.value);
        for (Connection c : this.OUTBOUND) {
            c.ff(d);
        }
    }

    public void push(double d) {
        this.value += d;
    }

    public double value() {
        return this.value;
    }

    public void reset() {
        this.value = 0.0;
    }

    @Override
    public String toString() {
        return String.format( "%s{Value: %.2g, Bias: %.2g, Connections: %d}", Knot.class.getSimpleName(), this.value, this.bias, this.OUTBOUND.size());
    }

    public double bias() {
        return bias;
    }

    public int id() {
        return this.LAYER_ID;
    }
}
