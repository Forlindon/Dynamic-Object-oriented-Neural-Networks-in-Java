package net.forlindon.dynamic.objectoriented.neural.networks;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.BaseConnection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.BaseKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.LinearLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.LinearSequentialLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.SequentialLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.MulTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SubTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class Test {

    public static void main(String[] args) {
        // This creates a neural network with a (2,3,1) structure
        // The knots are automatically connected via BaseConnections
        SequentialLayer seq = new LinearSequentialLayer(
                BaseConnection::new,
                i -> new LinearLayer(2, BaseKnot::new, i), // This factory creates a Linear Layer with 2 Knots of the Type BaseKnot
                i -> new LinearLayer(3, BaseKnot::new, i),
                i -> new LinearLayer(1, BaseKnot::new, i)
        );
        // This initializes the input layer and forwards everything
        for (int i = 0; i < 100; i++) {
            seq.forward(new double[]{1, 2});
            Tensor y = seq.get(2).getKNOTS().getFirst();
            Tensor target = new SimpleTensor(3);
            Tensor a = new SubTensor();
            a.activate(y, target);
            // This calculates the loss
            Tensor loss = new MulTensor();
            loss.activate(a, a);
            System.out.println(loss);
            loss.pushGrad(1);
            loss.derivative(a, a);
            a.derivative(y, target);
            // This calculates the gradient
            seq.backward();
            System.out.println(seq); // This shows the layers with values and gradients
            seq.getParameters().forEach(x -> x.grad *= 0.01); // similar to the learning rate
            seq.adjust(); // This subtracts the gradient from the values of the weights and bias
            System.out.println(seq); // Layers with adjusted values
            seq.clean(); // This cleans the knots and the parameters to be ready for the next iteration
            System.out.println(seq); // This shows the cleand layers
        }
    }

}
