package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

import javax.swing.*;
import java.awt.*;

public class CardPoleGame extends JPanel {

    public CardPoleGame(int size) {
        this.setSize(size,size);
    }

    double theta;

    public void updateTheta(double theta) {
        this.theta = theta;
    }

    @Override
    protected void paintComponent(Graphics g) {
        int cartY = getHeight() - 100;
        int cartWidth = 100;
        int cartHeight = 20;

        // int cartX = (int) x + getWidth() / 2;
        int cartX =  getWidth() / 2 - cartWidth / 2;

        g.setColor(Color.WHITE);
        g.fillRect(0,0,getWidth(),getHeight());
        g.setColor(Color.BLACK);
        g.fillRect(cartX, cartY, cartWidth, cartHeight);

        // Draw pole
        double poleX = cartX + cartWidth / 2.0;
        double poleY = cartY;
        double poleLen = 100;

        double endX = poleX + poleLen * Math.sin(theta);
        double endY = poleY - poleLen * Math.cos(theta);

        g.setColor(Color.RED);
        g.drawLine((int) poleX, (int) poleY, (int) endX, (int) endY);
    }

}
