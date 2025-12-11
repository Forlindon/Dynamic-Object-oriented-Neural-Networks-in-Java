package net.forlindon.dynamic.objectoriented.neat.experiments.tictactoe;

import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoTypeBuilder;
import net.forlindon.dynamic.objectoriented.neat.genetic.Species;

import java.util.List;

public class Test {

    public static void main(String[] args) {
        Engin engin = new Engin();
        for (int i = 0; i < 1000; i++) {
            engin.run();
            if (i % 100 == 99) {
                engin.evaluate();
                PhenoTypeBuilder phenoTypeBuilder = engin.getPhenoTypeBuilder();
                Environment environment = new Environment();
                List<Species> speciesList = engin.getSpeciesManager().getSpecies();
                for (int k = 0; k < speciesList.size()-1; k++) {
                    System.out.println("########## - " + k + " - ##########");
                    Species spA = speciesList.get(k);
                    Agent a = new Agent(spA.getFittest(), phenoTypeBuilder);
                    for (int j = k+1; j < speciesList.size(); j++) {
                        Species spB = speciesList.get(j);
                        Agent b = new Agent(spB.getFittest(), phenoTypeBuilder);
                        environment.play(a,b);
                        System.out.println();
                    }
                }
            }
        }
    }

}
