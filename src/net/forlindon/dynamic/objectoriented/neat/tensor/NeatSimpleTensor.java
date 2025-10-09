package net.forlindon.dynamic.objectoriented.neat.tensor;

import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class NeatSimpleTensor extends SimpleTensor {

    private final int inovNum;

    public NeatSimpleTensor(InnovationSource inov, double v) {
        super(v);
        this.inovNum = inov.getNext();
    }

    private NeatSimpleTensor(int inovNum, double v, double grad) {
        super(v);
        this.grad = grad;
        this.inovNum = inovNum;
    }

    @Override
    public String toString() {
        return String.format("%d: %s", this.inovNum, super.toString());
    }

    @Override
    public Tensor copy() {
        return new NeatSimpleTensor(this.inovNum,this.val,this.grad);
    }
}
