package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

import java.util.List;

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
            k.BIAS.resetGrad();
            k.IN.reset();
            k.OUT.reset();
            k.getConnections().forEach(Tensor::resetGrad);
        });
    }

    @Override
    public Layer copy() {
        Layer l = new BaseLayer(this.id());
        List<Knot> knots = this.getKNOTS();
        for (Knot k : knots) {
            l.add(k.copy());
        }
        return l;
    }
}
