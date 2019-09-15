/**
 * This class preprocesses a dataset. Dataset can be randomly partitioned into
 * k folds with or without normalization. A dataset must be partitioned before
 * being tested by KNN class and RBFNetwork class
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class ProcessData 
{
    String filePath = ""; // Path of the dataset
    String fileName = ""; // Name of the dataset
    ArrayList<String[]> fileArray = new ArrayList<String[]>(); // Stores data
    int numCols = 0;  // Number of columns in the dataset
    int fileSize = 0; // Number of instances in the dataset
    ArrayList<Double> mins = new ArrayList<Double>(); // For denormalization
    ArrayList<Double> maxs = new ArrayList<Double>(); // For denormalization
    // Stores the number of data points in each class
    HashMap<String, Integer> numClassPoints = new HashMap<String, Integer>();
    // Stores each partition (fold)
    ArrayList<ArrayList<String[]>> partitions = 
            new ArrayList<ArrayList<String[]>>();
    ArrayList<ArrayList<double[]>> XPartitions = 
            new ArrayList<ArrayList<double[]>>();
    // For regression tasks
    ArrayList<ArrayList<Double>> yValuePartitions = 
            new ArrayList<ArrayList<Double>>();
    // For classification tasks
    ArrayList<ArrayList<String>> yLabelPartitions = 
            new ArrayList<ArrayList<String>>();
    
    public ProcessData(String filePath)
    {
        this.filePath = filePath;
        switch (filePath)
        {
        case "ecoli.data":
            fileName = "Ecoli";
            break;
        case "segmentation.data":
            fileName = "Image Segmentation";
            break;
        case "machine.data":
            fileName = "Computer Hardware";
            break;
        case "forestfires.data":
            fileName = "Forest Fires";
            break;
        }
        // For datasets other than the 4 designated datasets
        if (fileName.equals(""))
        {
            fileName = filePath;
        }
    }
    
    /**
     * This method processes and stores a data file into an array
     * 
     * @param normalize suggests normalizing the data or not
     * @throws FileNotFoundException
     */
    public void process(boolean normalize) throws FileNotFoundException
    {
        File file = new File(filePath);
        Scanner reader = new Scanner(file);
        int numHeaderLines = 0;
        while (reader.hasNextLine())
        {
            if (fileName.equals("Ecoli")) // Handle the space-separated file
            {
                String[] line = reader.nextLine().split("\\s+");
                String[] instance = new String[line.length - 1];
                for (int i = 0; i < instance.length; i++)
                {// Ignore the first column "Sequence Name"
                    instance[i] = line[i + 1];
                }
                if (instance[instance.length - 1].equals("omL") 
                        || instance[instance.length - 1].equals("imL") 
                        || instance[instance.length - 1].equals("imS"))
                {// Remove three classes which have few examples
                    continue;
                }
                else
                {
                    fileArray.add(instance);
                }
            }
            else // Handle comma-separated files
            {
                String[] line = reader.nextLine().split(",");
                if (fileName.equals("Image Segmentation"))
                {
                    String[] instance = line;
                    for (int i = 0; i < instance.length - 1; i++)
                    {// Move the target column to the last position
                        String temp = instance[i];
                        instance[i] = instance[i + 1];
                        instance[i + 1] = temp;
                    }
                    fileArray.add(instance);
                    if (numHeaderLines < 5)
                    {// Remove the five-line header in segmentation.data
                        fileArray.remove(0);
                        numHeaderLines++;
                    }
                }
                else if (fileName.equals("Computer Hardware"))
                {
                    String[] instance = new String[line.length - 3];
                    for (int i = 0; i < instance.length; i++)
                    {// Ignore columns "Vendor Name", "Model Name", and "ERP"
                        instance[i] = line[i + 2];
                    }
                    fileArray.add(instance);
                }
                else if (fileName.equals("Forest Fires"))
                {
                    String[] instance = line;
                    String[] month = {"jan", "feb", "mar", "apr", "may", "jun", 
                            "jul", "aug", "sep", "oct", "nov", "dec"};
                    String[] day = {"mon", "tue", "wed", "thu", "fri", "sat", 
                            "sun"};
                    // Represent columns "month" and "day" as Roman numerals
                    for (int i = 0; i < month.length; i++)
                    {
                        if (instance[2].equals(month[i]))
                        {
                            instance[2] = Integer.toString(i + 1);
                        }
                    }
                    for (int j = 0; j < day.length; j++)
                    {
                        if (instance[3].equals(day[j]))
                        {
                            instance[3] = Integer.toString(j + 1);
                        }
                    }
                    fileArray.add(instance);
                    if (numHeaderLines == 0)
                    {// Remove the one-line header in forestfires.data
                        fileArray.remove(0);
                        numHeaderLines++;
                    }
                }
                else // For datasets other than the 4 designated datasets. The
                {    // dataset has to be comma-separated, with target column 
                     // at the last position and without a header
                    String[] instance = line;
                    fileArray.add(instance);
                }
            }
        }
        numCols = fileArray.get(0).length;
        fileSize = fileArray.size();
        if (normalize)
        {
            normalize();
        }
        reader.close();
    }
    
    /**
     * This method normalizes the dataset
     */
    public void normalize()
    {
        for (int i = 0; i < numCols - 1; i++)
        {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (String[] instance : fileArray)
            {
                if (Double.parseDouble(instance[i]) < min)
                {
                    min = Double.parseDouble(instance[i]);
                }
                if (Double.parseDouble(instance[i]) > max)
                {
                    max = Double.parseDouble(instance[i]);
                }
            }
            for (int j = 0; j < fileArray.size(); j++)
            {// Normalize all features to range from 0 to 1
                double original = Double.parseDouble(fileArray.get(j)[i]);
                if (max - min == 0)
                {// When all values in a column are the same, set them to 0.5
                    fileArray.get(j)[i] = Double.toString(0.5);
                }
                else
                {
                    double normalized = (original - min) / (max - min);
                    fileArray.get(j)[i] = Double.toString(normalized);
                }
            }
            mins.add(min);
            maxs.add(max);
        }
    } 
    
    /**
     * This method randomly partitions the dataset into k folds
     * 
     * @param k is the number of partitions (folds)
     * @param stratified suggests performing stratified partition or not
     * @param task is either regression or classification
     */
    public void kFold (int k, boolean stratified, String task)
    {
        if (stratified)
        {// Calculate the number of data points in each class
            for (int j = 0; j < fileArray.size(); j++)
            {
                if (numClassPoints.containsKey(fileArray.get(j)[numCols - 1]))
                {
                    int count = numClassPoints.get(fileArray.get(j)[numCols 
                                                                    - 1]);
                    count++;
                    numClassPoints.put(fileArray.get(j)[numCols - 1], count);
                }
                else
                {
                    numClassPoints.put(fileArray.get(j)[numCols - 1], 1);
                }
            }
        }
        for (int i = 0; i < k; i++) // Number of partitions
        {
            ArrayList<String[]> partition = new ArrayList<String[]>();
            if (stratified)
            {// Perform stratified split
                Iterator<Map.Entry<String, Integer>> it = 
                        numClassPoints.entrySet().iterator();
                while(it.hasNext())
                {// For each class, select enough points to current partition
                    Map.Entry<String, Integer> pair = it.next();
                    int classPoints = pair.getValue();
                    if (i == k - 1)
                    {// Add all remaining data points to the last partition
                        while (fileArray.size() != 0)
                        {
                            partition.add(fileArray.get(0));
                            fileArray.remove(0);
                        }
                        break;
                    }
                    // Make sure we have enough data points in each partition
                    int minPoints = classPoints / k;
                    if (minPoints < 1)
                    {
                        minPoints = 1;
                    }
                    int j = 0;
                    while(j < fileArray.size() && minPoints > 0)
                    {
                        if (pair.getKey().equals(fileArray.get(j)[numCols 
                                                                  - 1]))
                        {
                            partition.add(fileArray.get(j));
                            fileArray.remove(j);
                            minPoints--;
                        }
                        j++;
                    }
                }
            }
            else
            {// Perform random split
                if (i == k - 1)
                {// Add all remaining data points to the last partition
                    while (fileArray.size() != 0)
                    {
                        partition.add(fileArray.get(0));
                        fileArray.remove(0);
                    }
                }
                else
                {// Randomly select data points to add to the current partition
                    while(partition.size() < fileSize / k)
                    {
                        int selected = (int) (Math.random() 
                                * fileArray.size());
                        partition.add(fileArray.get(selected));
                        fileArray.remove(selected);
                    }
                }
            }
            partitions.add(partition);
        }
        convert(task);
    }
    
    /**
     * This method converts String arrays to double arrays
     * 
     * @param task it either regression or classification
     */ 
    public void convert(String task)
    {// "R" for regression task and "C" for classification task
        if (task.equalsIgnoreCase("R"))
        {
            for (ArrayList<String[]> partition : partitions)
            {
                ArrayList<double[]> XPartition = new ArrayList<double[]>();
                ArrayList<Double> yPartition = new ArrayList<Double>();
                
                for (String[] instance : partition)
                {
                    double[] X = new double[instance.length - 1];
                    double y = 0;
                    for (int i = 0; i < instance.length; i++)
                    {
                        if (i == instance.length - 1)
                        {
                            y = Double.parseDouble(instance[i]);
                        }
                        else
                        {
                            X[i] = Double.parseDouble(instance[i]);
                        }
                    }
                    XPartition.add(X);
                    yPartition.add(y);
                }
                XPartitions.add(XPartition);
                yValuePartitions.add(yPartition);
            }
        }
        else if (task.equalsIgnoreCase("C"))
        {
            for (ArrayList<String[]> partition : partitions)
            {
                ArrayList<double[]> XPartition = new ArrayList<double[]>();
                ArrayList<String> yPartition = new ArrayList<String>();
                
                for (String[] instance : partition)
                {
                    double[] X = new double[instance.length - 1];
                    String y = "";
                    for (int i = 0; i < instance.length; i++)
                    {
                        if (i == instance.length - 1)
                        {
                            y = instance[i];
                        }
                        else
                        {
                            X[i] = Double.parseDouble(instance[i]);
                        }
                    }
                    XPartition.add(X);
                    yPartition.add(y);
                }
                XPartitions.add(XPartition);
                yLabelPartitions.add(yPartition);
            }
        }
    }
}
