package net.forlindon.dynamic.objectoriented.neural.networks.connection;

import net.forlindon.dynamic.objectoriented.neat.TriFunction;
import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.BaseKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

public abstract class Connection extends Tensor {

    protected net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot src;
    protected net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot dest;

    public Connection(Knot src, Knot dest) {
        if (src == null || dest == null) throw new RuntimeException(String.format("Src and Dest shall not be null: %s %s", src, dest));
        this.src = src;
        this.dest = dest;
        this.val = (Math.random()-0.5)*0.1;
    }

    public double grad() {
        return this.grad;
    }

    public double weight() {
        return this.val;
    }

    public void ff() {
        this.dest.IN.activate(this.src.OUT, this);
    }

    public void fb() {
        this.dest.IN.derivative(this.src.OUT, this);
    }

    @Override
    public void activate(Tensor... args) {
    }

    @Override
    public void derivative(Tensor... args) {
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Connection c && this.dest == c.dest && this.src == c.src;
    }

    public void setSrc(Knot knot) {
        if (knot == null) throw new RuntimeException("Src shall not be null");
        this.src = knot;
    }

    public Knot src() {
        return this.src;
    }

    public void setDest(Knot dest) {
        if (dest == null) throw new RuntimeException("Dest shall not be null");
        this.dest = dest;
    }

    public Knot dest() {
        return this.dest;
    }

    public List<Connection> insert(Knot b, BiFunction<Knot, Knot, Connection> factory) {
        Knot a = src();
        Knot c = dest();
        a.connect(b, factory);
        b.connect(c, factory);
        return List.of(a.getConnections().getLast(),b.getConnections().getLast());
    }

    @Override
    public abstract Connection copy();
}
