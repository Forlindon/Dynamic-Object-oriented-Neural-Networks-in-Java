package net.forlindon.dynamic.objectoriented.neural.networks;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.BaseConnection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.*;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.*;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.loss.MSELoss;

import java.util.Arrays;

public class Test {

    public static void main(String[] args) {

        // This creates a neural network with a (2,3,1) structure
        // The knots are automatically connected via BaseConnections
        SequentialLayer seq = new LinearSequentialLayer(
                BaseConnection::new,
                i -> new LinearLayer(2, TanhKnot::new, i), // This factory creates a Linear Layer with 2 Knots of the Type BaseKnot
                i -> new LinearLayer(2, ReluKnot::new, i),
                i -> new LinearLayer(1, SigKnot::new, i)
        );
        // This initializes the input layer and forwards everything
        double[][] inputs = new double[][] {
                {0,0},
                {0,1},
                {1,0},
                {1,1}
        };
        double[] targets = new double[]{0,1,1,0};
        Tensor loss = new MSELoss();
        for (int i = 0; i < 100_000; i++) {
            int r = (int)(Math.random()*inputs.length);
            seq.forward(inputs[r]);
            Tensor y = seq.getLast().getKNOTS().getFirst().OUT;
            Tensor target = new SimpleTensor(targets[r]);
            loss.activate(y,target);
            loss.derivative(y,target);
            // This calculates the gradient
            seq.backward();
            if (i % 1000 == 0) {
                System.out.println(loss + " : " + y + " : " + Arrays.toString(inputs[r]));
                // System.out.println(seq.getParameters());
            }
            // if (i%100 == 0) System.out.println(seq); // This shows the layers with values and gradients
            seq.adjust(0.1); // This subtracts the gradient from the values of the weights and bias
            // System.out.println(seq); // Layers with adjusted values
            seq.clean(); // This cleans the knots and the parameters to be ready for the next iteration
            // System.out.println(seq.getParameters()); // This shows the cleand layers
        }
        for (double[] array : inputs) {
            seq.clean();
            seq.forward(array);
            double[] res = new double[1];
            seq.readValues(res);
            System.out.println(res[0] + " : " + Arrays.toString(array));
        }
    }

}
