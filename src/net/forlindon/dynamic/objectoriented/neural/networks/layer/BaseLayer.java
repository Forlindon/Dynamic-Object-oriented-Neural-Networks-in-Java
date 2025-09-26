package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

public class BaseLayer extends Layer {
    public BaseLayer(int id) {
        super(id);
    }

    @Override
    public void forward() {
        this.PARAMETERS.forEach(Knot::pop);
    }

    @Override
    public void backward() {
        this.PARAMETERS.forEach(Knot::backward);
    }
}
