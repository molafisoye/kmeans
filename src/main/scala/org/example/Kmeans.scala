package org.example

import java.io.FileWriter
import scala.io.Source
import scala.util.Random

object Kmeans {
  def main(args: Array[String]): Unit = {
    val inputFileName = "iris.txt"
    val iterations = args(0).toInt
    val groups = args(1).toInt
    val outputDir = args(2)

    val dataPoints = Source.fromResource(inputFileName).getLines().map { line =>
      val splitLines = line.split(",")
      Point(splitLines(0).toDouble, splitLines(1).toDouble, splitLines(2).toDouble, splitLines(3).toDouble)
    }.toList

    val fw = new FileWriter(outputDir + "/output.csv", true)
    val km = getKmeans(dataPoints, groups, iterations)

    km.indices.foldLeft(km){ case (clusterGroup, index) =>
      fw.write(s"clustering, ${index + 1}\n")
      clusterGroup.last.foreach{cluster =>
        fw.write(s"distortion, ${cluster.distortion}\n")
        fw.write(s"centroid, ${cluster.centroid.sepalLength}, ${cluster.centroid.sepalWidth}, ${cluster.centroid.petalLength}, ${cluster.centroid.petalWidth}\n")
        cluster.points.foreach(point => fw.write(s"point, ${point.sepalLength}, ${point.sepalWidth}, ${point.petalLength}, ${point.petalWidth}\n"))
        fw.write("\n")
      }
      clusterGroup.init
    }
  }

  def getCentroid(dataPoints: List[Point], numberOfClusters: Int): List[Cluster] = {
    Random.shuffle(dataPoints).take(numberOfClusters).map(Cluster(_))
  }

  def groupPoints(clusters: List[Cluster], points: List[Point]): List[Cluster] = {
    val pointsByCluster = points.groupBy(point => clusters.minBy(_.centroid.distanceTo(point)))
    clusters.map(cluster => cluster.copy(points = pointsByCluster.getOrElse(cluster, Nil)))
  }

  def recenter(clusters: List[Cluster]): List[Cluster] = {
    clusters.map { cluster =>
      val sLength = cluster.points.map(_.sepalLength).sum / cluster.points.size
      val sWidth = cluster.points.map(_.sepalWidth).sum / cluster.points.size
      val pLength = cluster.points.map(_.petalLength).sum / cluster.points.size
      val pWidth = cluster.points.map(_.petalWidth).sum / cluster.points.size
      val newCentroid = Point(sLength, sWidth, pLength, pWidth)
      val newDistortion = newCentroid.getDistortion(cluster.points)
      cluster.copy(centroid = Point(sLength, sWidth, pLength, pWidth), distortion = newDistortion )
    }
  }

  def getKmeans(points: List[Point], n: Int, clustering: Int): List[List[Cluster]] = {
    val initClusters = List(getCentroid(points, n))

    (1 to clustering).foldLeft(initClusters) { (cluster, _) =>
      val groupedPoints = groupPoints(cluster.head, points)
      val reCentered = recenter(groupedPoints)
      reCentered +: cluster
    }
  }
}

case class Point(sepalLength: Double, sepalWidth: Double, petalLength: Double, petalWidth: Double) {
  def distanceTo(other: Point): Double = {
    math.sqrt(math.pow(this.sepalLength - other.sepalLength, 2) + math.pow(this.sepalWidth - other.sepalWidth, 2) +
      math.pow(this.petalLength - other.petalLength, 2) + math.pow(this.petalWidth - other.petalWidth, 2))
  }

  def getDistortion(others: List[Point]): Double = {
    others.map(point => distanceTo(point)).sum
  }
}

case class Cluster(centroid: Point, distortion: Double = 0.0, points: List[Point] = Nil)

case class ClusterBatch(clusters: List[Cluster])