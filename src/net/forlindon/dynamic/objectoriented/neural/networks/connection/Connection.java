package net.forlindon.dynamic.objectoriented.neural.networks.connection;

import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public abstract class Connection extends Tensor {

    protected Knot src;
    protected Knot dest;

    public Connection(Knot src,Knot dest) {
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

    public void setSrc(BaseNeatKnot baseNeatKnot) {
        this.src = baseNeatKnot;
    }

    public BaseNeatKnot src() {
        return (BaseNeatKnot) this.src;
    }

    public void setDest(BaseNeatKnot dest) {
        this.dest = dest;
    }

    public BaseNeatKnot dest() {
        return (BaseNeatKnot) this.dest;
    }
}
