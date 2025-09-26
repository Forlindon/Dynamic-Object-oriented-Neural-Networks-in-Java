package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class BaseLayer extends Layer {
    public BaseLayer(int id) {
        super(id);
    }

    @Override
    public void forward() {
        this.KNOTS.forEach(Knot::pop);
    }

    @Override
    public void backward() {
        this.KNOTS.forEach(Knot::backward);
    }

    @Override
    public void clean() {
        this.KNOTS.forEach(k -> {
            k.reset();
            k.BIAS.resetGrad();
            k.getConnections().forEach(Tensor::resetGrad);
        });
    }
}
