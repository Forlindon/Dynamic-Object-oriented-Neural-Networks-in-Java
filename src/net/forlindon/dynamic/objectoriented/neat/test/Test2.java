package net.forlindon.dynamic.objectoriented.neat.test;

import net.forlindon.dynamic.objectoriented.neat.NeatEngine;
import net.forlindon.dynamic.objectoriented.neat.knot.SigNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;
import net.forlindon.dynamic.objectoriented.neat.genetic.SpeciesManager;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetWrapper;
import net.forlindon.dynamic.objectoriented.neat.visuals.NetworkDisplay;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.TanhTensor;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class Test2 {

    static {
        SpeciesManager.setDefaultActivation(TanhTensor::new);
    }

    public static NeatEngine neatEngin = new NeatEngine(2,1,100, Test2::eval, 1, SigNeatKnot::new);

    static double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };

    static double[] targets = {0, 1, 1, 0};

    static double[] out = new double[1];

    static NetWrapper netWrapper = new NetWrapper(null);
    static {
        neatEngin.update(netWrapper);
    }

    public static void main(String[] args) {

        EventQueue.invokeLater(() -> {
            JFrame jFrame = new JFrame("NET DISPLAY");
            jFrame.setSize(500,500);
            jFrame.add(new NetworkDisplay(netWrapper));
            jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            jFrame.setVisible(true);
        });

        new Thread(() -> {
            Genome fittest = null;
            for (int i = 0; i < 300; i++) {
                fittest = neatEngin.run();
                if (i % 10 == 0) {
                    neatEngin.update(netWrapper);
                }
                if (fittest.getFitness() > 3.9) {
                    break;
                }
            }
            System.out.println("FIN!");
            neatEngin.update(netWrapper);
            System.out.println(fittest.getFitness());
            showResults(fittest);
        }).start();

    }

    public static double eval(Genome g) {
        PhenoType phenoType = neatEngin.getPhenoTypeBuilder().build(g);
        double lossum = 0;
        for (int i = 0; i < inputs.length; i++) {
            phenoType.forward(inputs[i], out);
            double delta = Math.pow(out[0] - targets[i], 2);
            lossum += delta;
        }
        return 4.0 - lossum;
    }

    public static void showResults(Genome g) {
        PhenoType phenoType = neatEngin.getPhenoTypeBuilder().build(g);
        for (double[] input : inputs) {
            phenoType.forward(input, out);
            System.out.printf("%s: %g\n", Arrays.toString(input), out[0]);
        }
    }


}
