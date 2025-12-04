package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

import net.forlindon.dynamic.objectoriented.neat.experiments.Pair;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;

import java.awt.*;

public class Test {

    public static void main(String[] args) {
        CardPoleEngin cardPoleEngin = new CardPoleEngin();
        Genome run = null;
        for (int i = 0; i < 100; i++) {
            run = cardPoleEngin.run();
            if (run.getFitness() > 1900) break;
        }
        System.out.println("Finished");
        Genome finalRun = run;
        EventQueue.invokeLater(() -> {
            CardPoleFrame cardPoleFrame = new CardPoleFrame(new Agent(new Pair<>(finalRun, cardPoleEngin.getPhenoTypeBuilder().build(finalRun))));
        });
    }

}
