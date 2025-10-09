package net.forlindon.dynamic.objectoriented.neural.networks.connection;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class BaseConnection extends Connection {

    public BaseConnection(Knot src, Knot dest) {
        super(src, dest);
    }

    public BaseConnection(Knot src, Knot dest, double w) {
        this(src,dest,w,0);
    }

    public BaseConnection(Knot src, Knot dest, double w, double grad) {
        super(src,dest);
        this.val = w;
        this.grad = grad;
    }

    @Override
    public Tensor copy() {
        return new BaseConnection(this.src,this.dest,this.val,this.grad);
    }
}
