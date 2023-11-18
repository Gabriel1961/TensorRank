import * as tf from "@tensorflow/tfjs";
import { DatasetData } from "../../datamodel";

export const evalModel = async (dataset: DatasetData, model: tf.LayersModel) => {  
  console.log('started fitting')
  await model.fit(dataset.trainX, dataset.trainY, { epochs: 100, shuffle: true })
  const result = model.evaluate(dataset.trainX, dataset.trainY, {}) as any
  const acc = result[1].arraySync()
  return acc
}


