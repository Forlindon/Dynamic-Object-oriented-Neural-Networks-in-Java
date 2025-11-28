package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.MulTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

public abstract class Knot {

    protected final List<Connection> OUTBOUND;
    private final int LAYER_ID;
    public Tensor BIAS;
    public Tensor IN = new MulTensor();
    public Tensor OUT;

    public Knot(int id) {
        this.LAYER_ID = id;
        this.OUTBOUND = new ArrayList<>();
        this.OUT = getActivationTensor();
        this.BIAS = getBIAS();
    }

    public abstract Tensor getBIAS();

    public void connect(Knot other, BiFunction<Knot, Knot, Connection> factory) {
        if (other.LAYER_ID == this.LAYER_ID) throw new IllegalArgumentException("Invalid LAYER_ID");
        Connection c = factory.apply(this, other);
        if (this.OUTBOUND.contains(c)) throw new IllegalArgumentException("No duplicate connections");
        this.OUTBOUND.add(c);
    }

    public abstract Tensor getActivationTensor();

    public void pop() {
        this.OUT.activate(this.IN,this.BIAS);
        forward();
    }

    private void forward() {
        this.OUTBOUND.forEach(Connection::ff);
    }

    public void backward() {
        this.OUTBOUND.forEach(Connection::fb);
        this.OUT.derivative(this.IN,this.BIAS);
    }

    @Override
    public String toString() {
        return String.format( "%s{IN: %s, OUT: %s, BIAS: %s, Connections: %s}", this.getClass().getSimpleName(), this.IN, this.OUT, this.BIAS, this.OUTBOUND);
    }

    public double bias() {
        return BIAS.val;
    }

    public int id() {
        return this.LAYER_ID;
    }

    public void reset() {
        this.IN.reset();
        this.OUT.reset();
    }

    public List<Connection> getConnections() {
        return new ArrayList<>(this.OUTBOUND);
    }

    public void add(Connection c) {
        if (!this.OUTBOUND.contains(c)) {
            this.OUTBOUND.add(c);
        }
    }

    public boolean isConnectedTo(Knot o) {
        return !this.getConnections().stream().map(Connection::dest).filter(knot -> knot.equals(o)).toList().isEmpty();
    }

    public abstract Knot copy();

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Knot knot)) return false;
        return LAYER_ID == knot.LAYER_ID && Objects.equals(OUTBOUND, knot.OUTBOUND) && Objects.equals(BIAS, knot.BIAS) && Objects.equals(IN, knot.IN) && Objects.equals(OUT, knot.OUT);
    }

    @Override
    public int hashCode() {
        return Objects.hash(OUTBOUND, LAYER_ID, BIAS, IN, OUT);
    }
}
