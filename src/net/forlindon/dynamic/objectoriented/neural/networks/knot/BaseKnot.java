package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.SimpleActivationTensor;

import java.util.List;

public class BaseKnot extends Knot{
    public BaseKnot(int id) {
        super(id);
    }

    protected BaseKnot(int id, Tensor bias, Tensor act, List<Connection> connections) {
        super(id);
        this.BIAS = bias;
        this.OUT = act;
        this.OUTBOUND.addAll(connections);
        connections.forEach(connection -> connection.setSrc(this));
    }

    @Override
    public Tensor getBIAS() {
        return new SimpleTensor(Math.random()*0.1);
    }

    @Override
    public Tensor getActivationTensor() {
        return new SimpleActivationTensor();
    }

    @Override
    public Knot copy() {
        return new BaseKnot(this.id(), this.BIAS.copy(), this.OUT.copy(), this.OUTBOUND.stream().map(Connection::copy).toList());
    }
}
