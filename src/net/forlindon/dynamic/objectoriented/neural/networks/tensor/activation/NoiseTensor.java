package net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class NoiseTensor extends SimpleActivationTensor {

    double p;
    double mask;

    public NoiseTensor(double p) {
        this.p = p;
    }

    public NoiseTensor() {
        this(0.5);
    }

    @Override
    public void activate(Tensor... args) {
        super.activate(args);
        this.mask = Math.random() < 0.5 ? 0 : 1;
        this.val *= mask;
    }

    @Override
    public void derivative(Tensor... args) {
        this.grad*=mask*(args[0].val+args[1].val);
        super.derivative(args);
    }
}
