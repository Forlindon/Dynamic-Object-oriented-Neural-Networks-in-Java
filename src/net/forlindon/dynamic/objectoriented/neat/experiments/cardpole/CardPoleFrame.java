package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class CardPoleFrame extends JFrame {

    public CardPoleGame cardPoleGame;

    Agent agent;

    CardPoleEnvironment cardPoleEnvironment = new CardPoleEnvironment();

    public CardPoleFrame(Agent agent) {
        this.agent = agent;
        this.setSize(400,400);
        cardPoleGame = new CardPoleGame(400);
        this.setLayout(new GridLayout());
        this.add(cardPoleGame);
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        this.setResizable(false);
        this.setVisible(true);

        Timer t = new Timer(16, this::tick);
        t.start();
    }

    public void tick(ActionEvent e) {
        cardPoleEnvironment.runAgent(this.agent);
        this.cardPoleGame.updateTheta(this.agent.state[0]);
        if (this.agent.done) this.agent.reset();
        repaint();
    }


}
