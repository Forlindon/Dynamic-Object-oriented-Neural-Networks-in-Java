package net.forlindon.dynamic.objectoriented.neural.networks.connection;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public abstract class Connection extends Tensor {

    Knot src;
    Knot dest;

    public Connection(Knot src,Knot dest) {
        this.src = src;
        this.dest = dest;
        this.val = Math.random()-0.5;
    }

    public double grad() {
        return this.grad;
    }

    public double weight() {
        return this.val;
    }

    public void ff() {
        this.dest.activate(this.src, this);
    }

    public void fb() {
        this.dest.derivative(this.src, this);
    }

    @Override
    public void activate(Tensor... args) {
    }

    @Override
    public void derivative(Tensor... args) {
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Connection c && this.dest == c.dest;
    }
}
