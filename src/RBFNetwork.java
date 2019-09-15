/**
 * This class implements the RBF Network Algorithm. It handles both regression
 * and classification tasks. The test() method takes in datasets that are 
 * already preprocessed by the ProcessData class. The inner class Node 
 * implements the hidden node
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class RBFNetwork 
{
    class Node
    {
        double[] center;   // Center of the hidden node
        double sigma = 0;  // Spread of the hidden node
        double output = 0; // Output of the hidden node
        ArrayList<Double> weights = new ArrayList<Double>(); // Stores weights
        
        public Node(double[] center)
        {
            this.center = center;
        }
        
        /**
         * This class calculates the RBF value for a query point
         * 
         * @param X is the query point
         */
        public void RBF(double[] X)
        {
            double distance = 0;
            for (int i = 0; i < X.length; i++)
            {
                distance += Math.pow((X[i] - center[i]), 2);
            }
            output = Math.exp(((-1.0) / (2.0 * sigma * sigma)) * distance);
        }
        
        /**
         * This class sets the value of spread
         * 
         * @param sigma = spread
         */
        public void setSigma(double sigma)
        {
            this.sigma = sigma;
        }
    }
    
    String task = ""; // Type of the task
    ArrayList<ArrayList<double[]>> XPartitions = 
            new ArrayList<ArrayList<double[]>>();
    ArrayList<double[]> XTrain = new ArrayList<double[]>();
    ArrayList<double[]> XTest = new ArrayList<double[]>();
    // For regression tasks
    ArrayList<ArrayList<Double>> yValuePartitions = 
            new ArrayList<ArrayList<Double>>();
    ArrayList<Double> yValueTrain = new ArrayList<Double>();
    ArrayList<Double> yValueTest = new ArrayList<Double>();
    // For classification tasks
    ArrayList<ArrayList<String>> yLabelPartitions = 
            new ArrayList<ArrayList<String>>();
    ArrayList<String> yLabelTrain = new ArrayList<String>();
    ArrayList<String> yLabelTest = new ArrayList<String>();
    
    ArrayList<Double> errors = new ArrayList<Double>(); // Error for each fold
    ArrayList<Double> accuracies = new ArrayList<Double>(); // Accuracy / fold
    ArrayList<Node> nodes = new ArrayList<Node>(); // Hidden nodes
    double eta = 0; // Learning rate
    ArrayList<String> classNames = new ArrayList<String>(); // Names of classes
    double[] classOutputs; // Output for each class
    double[] logisticClassOutputs; // Logistic output for each class
    
    public RBFNetwork(String task)
    {// "R" for regression task and "C" for classification task
        this.task = task;
    }
    
    /**
     * This method handles regression and classification tasks separately
     * 
     * @param data is the k-fold partitioned data processed by ProcessData
     * @param eta is the learning rate for Gradient Descent
     * @param threshold is the convergence threshold for training
     * @return average MSE or average accuracy from cross validation
     */
    public double test(ProcessData data, double eta, double threshold)
    {
        this.XPartitions = data.XPartitions;
        this.yValuePartitions = data.yValuePartitions;
        this.yLabelPartitions = data.yLabelPartitions;
        this.eta = eta;
        double score = 0;
        
        if (task.equalsIgnoreCase("R"))
        {// Return the average error from cross validation
            score = RBFRegressor(threshold);
        }
        else if (task.equalsIgnoreCase("C"))
        {// Return the average accuracy from cross validation
            for (ArrayList<String> partition : yLabelPartitions)
            { 
                for (String label : partition)
                {
                    if (!classNames.contains(label))
                    {
                        classNames.add(label);
                    }
                }
            }
            classOutputs = new double[classNames.size()];
            logisticClassOutputs = new double[classNames.size()];
            score = RBFClassifier(threshold);
        }
        return score;
    }
    
    /**
     * This method handles regression tasks
     * 
     * @param threshold is the convergence threshold for training
     * @return average MSE from cross validation
     */
    public double RBFRegressor(double threshold)
    {
        for (int i = 0; i < XPartitions.size(); i++)
        {// Perform RBF with each train-test pair
            XTest = XPartitions.get(i);
            yValueTest = yValuePartitions.get(i);
            for (int j = 0; j < XPartitions.size(); j++)
            {
                if (j != i)
                {
                    for (double[] X : XPartitions.get(j))
                    {
                        XTrain.add(X);
                    }
                    for (double yValue : yValuePartitions.get(j))
                    {
                        yValueTrain.add(yValue);
                    }
                }
            }
            // Train the RBF regressor
            double tenPercent = XTrain.size() * 0.1;
            for (int p = 0; p < tenPercent; p++)
            {// Establish 1 hidden node for 10% of the training set
                int random = (int) Math.random() * XTrain.size();
                nodes.add(new Node(XTrain.get(random)));
                XTrain.remove(random);
                yValueTrain.remove(random);
            }
            // Calculate spread(sigma) using the max distance among nodes
            double maxDistance = Double.MIN_VALUE;
            for (int f = 0; f < nodes.size() - 1; f++)
            {
                for (int g = f + 1; g < nodes.size(); g++)
                {
                    double centerDistance = getDistance(nodes.get(f).center,
                            nodes.get(g).center);
                    if (centerDistance > maxDistance)
                    {
                        maxDistance = centerDistance;
                    }
                }
            }
            double sigma = maxDistance / Math.sqrt(2 * nodes.size());
            for (Node node : nodes)
            {// Initialize random weights and set sigma
                node.weights.add(Math.random());
                node.setSigma(sigma);
            }
            boolean converge = false;
            double previousError = Double.MAX_VALUE;
            while (!converge)
            {
                ArrayList<Double> yPredicts = new ArrayList<Double>();
                for (int q = 0; q < XTrain.size(); q++)
                {// Adjust the weights while going through the training set
                    for (Node node : nodes)
                    {// Calculate output of each hidden node
                        node.RBF(XTrain.get(q));
                    }
                    double finalOutput = 0;
                    for (Node node : nodes)
                    {// Calculate the final output with current weights
                        finalOutput += (node.weights.get(0) * node.output);
                    }
                    yPredicts.add(finalOutput);
                    for (Node node : nodes)
                    {// Update weights using gradient descent
                        double gradient = (finalOutput - yValueTrain.get(q)) 
                                * node.output;
                        node.weights.set(0, node.weights.get(0) - eta 
                                * gradient);
                    }
                }
                double error = 0;
                for (int r = 0; r < yPredicts.size(); r++)
                {
                    error += Math.pow((yPredicts.get(r) - yValueTrain.get(r)),
                            2);
                }
                error /= yPredicts.size();
                
                System.out.println("Adjusting training error ... " + error);
                
                if (Math.abs(error - previousError) < threshold 
                        * previousError)
                {// Stop when change in error is less than "threshold" percent
                    converge = true;
                }
                else
                {
                    previousError = error;
                }
            }
            ArrayList<Double> yPredicts = new ArrayList<Double>();
            for (int m = 0; m < XTest.size(); m++)
            {// For each test sample, make a prediction about its target value
                for (Node node : nodes)
                {// Calculate output of each hidden node
                    node.RBF(XTest.get(m));
                }
                double finalOutput = 0;
                for (Node node : nodes)
                {// Calculate the final output with current weights
                    finalOutput += (node.weights.get(0) * node.output);
                }
                yPredicts.add(finalOutput);
            }
            double error = 0;
            for (int t = 0; t < yPredicts.size(); t++)
            {// Calculate the mean squared error for this fold
                error += Math.pow((yPredicts.get(t) - yValueTest.get(t)), 2);
            }
            error /= yPredicts.size();
            errors.add(error);
            
            // For demonstration purpose
            System.out.println("-------------------------------");
            System.out.print("Weights: [ ");
            for (Node node : nodes)
            {
                System.out.print(node.weights.get(0) + " ");
            }
            System.out.println("]");
            System.out.println("Test error: " + error);
            System.out.println("-------------------------------");
            
            XTrain.clear();
            yValueTrain.clear();
        }
        // Calculate average error from all k folds
        double averageError = 0;
        for (Double error : errors)
        {
            averageError += error;
        }
        averageError /= XPartitions.size();
        errors.clear();
        
        // For demonstration purpose
        System.out.println("Average error = " + averageError);
        
        return averageError;
    }
    
    /**
     * This method calculates the Euclidean distance between two points
     * 
     * @param x1 is the first point
     * @param x2 is the second point
     * @return the Euclidean distance
     */
    public double getDistance(double[] x1, double[] x2)
    {
        double distance = 0;
        for (int i = 0; i < x1.length; i++)
        {
            distance += Math.pow((x1[i] - x2[i]), 2);
        }
        return Math.sqrt(distance);
    }
    
    /**
     * This method handles classification tasks
     * 
     * @param threshold is the convergence threshold for training
     * @return average classification accuracy from cross validation
     */
    public double RBFClassifier(double threshold)
    {
        for (int i = 0; i < XPartitions.size(); i++)
        {// Perform RBF with each train-test pair
            XTest = XPartitions.get(i);
            yLabelTest = yLabelPartitions.get(i);
            for (int j = 0; j < XPartitions.size(); j++)
            {
                if (j != i)
                {
                    for (double[] X : XPartitions.get(j))
                    {
                        XTrain.add(X);
                    }
                    for (String yLabel : yLabelPartitions.get(j))
                    {
                        yLabelTrain.add(yLabel);
                    }
                }
            }
            // Train the RBF classifier
            double tenPercent = XTrain.size() * 0.1;
            for (int p = 0; p < tenPercent; p++)
            {// Establish 1 hidden node for 10% of the training set
                int random = (int) Math.random() * XTrain.size();
                nodes.add(new Node(XTrain.get(random)));
                XTrain.remove(random);
                yLabelTrain.remove(random);
            }
            // Calculate spread(sigma) using the max distance among nodes
            double maxDistance = Double.MIN_VALUE;
            for (int f = 0; f < nodes.size() - 1; f++)
            {
                for (int g = f + 1; g < nodes.size(); g++)
                {
                    double centerDistance = getDistance(nodes.get(f).center, 
                            nodes.get(g).center);
                    if (centerDistance > maxDistance)
                    {
                        maxDistance = centerDistance;
                    }
                }
            }
            double sigma = maxDistance / Math.sqrt(2 * nodes.size());
            for (Node node : nodes)
            {// Set sigma
                node.setSigma(sigma);
            }
            for (int h = 0; h < classNames.size(); h++)
            {// Initialize weights for each hidden node. One weight per class
                for (Node node : nodes)
                {
                    node.weights.add(Math.random() / 2);
                }
            }
            double trainingAccuracy = Double.MIN_VALUE;
            while (trainingAccuracy < threshold)
            {// Continue adjusting the weight until accuracy exceeds 0.75
                double correctPrediction = 0; // Number of correct predictions
                for (int q = 0; q < XTrain.size(); q++)
                {// Adjust the weights while going through the training set
                    for (Node node : nodes)
                    {// Calculate output of each hidden node
                        node.RBF(XTrain.get(q));
                    }
                    for (int r = 0; r < classNames.size(); r++)
                    {// Calculate the output with current weight for each class
                        double classOutput = 0;
                        for (Node node : nodes)
                        {
                            classOutput += (node.weights.get(r) * node.output);
                        }
                        classOutputs[r] = classOutput;
                    }
                    for (int s = 0; s < classNames.size(); s++)
                    {// Transform the outputs using the sigmoid function
                        logisticClassOutputs[s] = 1 / (1 + Math.exp((-1.0) 
                                * classOutputs[s]));
                    }
                    double maxProbability = Double.MIN_VALUE;
                    int finalOutput = 0;
                    for (int t = 0; t < classNames.size(); t++)
                    {// Prediction would be the class with highest probability
                        if (logisticClassOutputs[t] > maxProbability)
                        {
                            maxProbability = logisticClassOutputs[t];
                            finalOutput = t;
                        }
                    }
                    if (classNames.get(finalOutput).equals(yLabelTrain.get(q)))
                    {
                        correctPrediction++;
                    }
                    for (int u = 0; u < classNames.size(); u++)
                    {// Update weights for each class using gradient descent
                        int trueProbability = 0;
                        if (classNames.get(u).equals(yLabelTrain.get(q)))
                        {// Weights are updated according to the target class
                            trueProbability = 1;
                        }
                        for (Node node : nodes)
                        {// Increase weight if target class, decrease otherwise
                            double gradient = (trueProbability 
                                    - logisticClassOutputs[u]) 
                                    * logisticClassOutputs[u] 
                                            * (1 - logisticClassOutputs[u]) 
                                            * node.output;
                            node.weights.set(u, node.weights.get(u) 
                                    + eta * gradient);
                        }
                    }
                }
                trainingAccuracy = correctPrediction / XTrain.size();  
                
                // For demonstration purpose
                System.out.println("Adjusting training accuracy ... " 
                + trainingAccuracy);
            }
            ArrayList<String> yPredicts = new ArrayList<String>();
            for (int m = 0; m < XTest.size(); m++)
            {// For each test sample, make a prediction about its target label
                for (Node node : nodes)
                {// Calculate output of each hidden node
                    node.RBF(XTest.get(m));
                }
                for (int r = 0; r < classNames.size(); r++)
                {// Calculate the output with current weight for each class
                    double classOutput = 0;
                    for (Node node : nodes)
                    {
                        classOutput += (node.weights.get(r) * node.output);
                    }
                    classOutputs[r] = classOutput;
                }
                for (int s = 0; s < classNames.size(); s++)
                {// Transform the outputs using the sigmoid function
                    logisticClassOutputs[s] = 1 / (1 + Math.exp((-1.0) 
                            * classOutputs[s]));
                }
                double maxProbability = Double.MIN_VALUE;
                int finalOutput = 0;
                for (int t = 0; t < classNames.size(); t++)
                {// Prediction would be the class with highest probability
                    if (logisticClassOutputs[t] > maxProbability)
                    {
                        maxProbability = logisticClassOutputs[t];
                        finalOutput = t;
                    }
                }
                yPredicts.add(classNames.get(finalOutput));
            }
            double correctPrediction = 0;
            for (int j = 0; j < yPredicts.size(); j++)
            {// Calculate the accuracy for this fold
                if (yPredicts.get(j).equals(yLabelTest.get(j)))
                {
                    correctPrediction++;
                }
            }
            double accuracy = correctPrediction / yPredicts.size();
            accuracies.add(accuracy);
            
            // For demonstration purpose
            System.out.println("-------------------------------");
            System.out.print("Weights for one class: [ ");
            for (Node node : nodes)
            {
                System.out.print(node.weights.get(0) + " ");
            }
            System.out.println("]");
            System.out.println("Test accuracy: " + accuracy);
            System.out.println("-------------------------------");
            
            XTrain.clear();
            yLabelTrain.clear();
        }
        // Calculate average classification accuracy from all k folds
        double averageAccuracy = 0;
        for (Double accuracy : accuracies)
        {
            averageAccuracy += accuracy;
        }
        averageAccuracy /= XPartitions.size();
        accuracies.clear();
        
        // For demonstration purpose
        System.out.println("Average accuracy = " + averageAccuracy);
        
        return averageAccuracy;
    }
}
