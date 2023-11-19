import { Timestamp, addDoc, collection, doc, getDoc, getDocs, query, where } from "firebase/firestore";
import { Dataset, DatasetData, Solution, Task } from "../../datamodel";
import { db, storage } from "../../firebaseConfig";
import * as tf from "@tensorflow/tfjs"
import { ref, getBytes } from "firebase/storage";
import { User } from "firebase/auth";


export const fetchTaskAfterTitle = async (taskTitle: string) => {
  const tasksCollection = collection(db, "tasks")
  const tasksDocs = (await getDocs(query(tasksCollection, where("title", "==", taskTitle)))).docs
  if (tasksDocs.length) {
    return { ...tasksDocs[0].data(), id: tasksDocs[0].id } as Task
  }
  return null
}

function shuffleArray(array: any[][]) {
  var mrng = new Math.seedrandom("start")

  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(mrng() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function parseCSVToTensor(csvString: string, dataHeaders: string[], lables: string[]) {
  // Split the CSV string into an array of lines
  const lines = csvString.split('\n');

  // Initialize an array to store the parsed data
  const parsedTable: number[][] = [];
  const initialTable: string[][] = []


  const classes:Map<string,number>[] = []
  for(let i=0;i<dataHeaders.length;i++)
    classes.push(new Map<string,number>())

  // Iterate over each line in the CSV
  for (let i = 0; i < lines.length; i++) {
    // Split the line into an array of values using commas
    const values = lines[i].split(',');
    // Add the array of values to the result array
    if (values.length && values[0]) {
      const newValues: number[] = []
      values.forEach((val,col) => {
        const parsed = parseFloat(val)
        if (Number.isNaN(parsed)) {
          if (classes[col].has(val)) {
            newValues.push(classes[col].get(val)!)
          }
          else {
            classes[col].set(val, classes[col].size)
            newValues.push(classes[col].size - 1)
          }
        }
        else {
          newValues.push(parsed)
        }
      })
      parsedTable.push(newValues)
      initialTable.push(values)
    }
  }

  const rows = parsedTable.length;
  const cols = parsedTable[0].length
  shuffleArray(parsedTable)
  shuffleArray(initialTable)

  const dataHead = initialTable.slice(0, Math.min(20, rows))

  // get indexes of each type of column 
  const lableIdxs: number[] = []
  const featureIdxs: number[] = []
  lables.forEach(lable => lableIdxs.push(dataHeaders.indexOf(lable)))
  lableIdxs.sort()
  for (let i = 0; i < cols; i++)
    if (lableIdxs.indexOf(i) == -1) {
      featureIdxs.push(i)
    }

  // separate data 
  const lableData: number[][] = []
  const featureData: number[][] = []

  for (let i = 0; i < rows; i++) {
    lableData.push([])
    featureData.push([])
  }

  lableIdxs.forEach((lableIdx) => {
    for (let i = 0; i < rows; i++) {
      lableData[i].push(parsedTable[i][lableIdx])
    }
  })

  featureIdxs.forEach((featureIdx) => {
    for (let i = 0; i < rows; i++) {
      featureData[i].push(parsedTable[i][featureIdx])
    }
  })

  const splitRow = Math.floor(rows * .8)
  const X = tf.tensor2d(featureData)
  let Y

  // only supports one hot for a single lable 
  if (classes[lableIdxs[0]].size) {
    Y = tf.oneHot(tf.cast(tf.tensor(lableData).squeeze(), 'int32'), classes[lableIdxs[0]].size)
  }
  else
    Y = tf.tensor2d(lableData)


  const trainX = X.slice(0, splitRow)
  const testX = X.slice(splitRow)
  const trainY = Y.slice(0, splitRow)
  const testY = Y.slice(splitRow)

  // Convert the result array to a 2D tensor
  return { dataHead, testX, testY, trainX, trainY, headers: dataHeaders } as DatasetData;
}

export const fetchTaskData = async (task: Task) => {
  const datasetDoc = await getDoc(doc(db, "datasets", task.datasetId))
  const dataset = { ...datasetDoc.data(), id: datasetDoc.id } as Dataset
  const dataBytes = await getBytes(ref(storage, `datasets/${dataset.name}/data`))
  const parsedData = parseCSVToTensor(new TextDecoder().decode(dataBytes), dataset.headers, dataset.lables)
  return {parsedData,dataset}
}

export const uploadSolution = (accuracy: number, taskId: string, user: User, code: string) => {
  const solution: Solution = {
    id: "",
    score: accuracy,
    taskId,
    sourceCode: code,
    dateCreated: Timestamp.now(),
    author: {
      name: user.displayName ?? user.email ?? "",
      id: user.uid
    }
  }

  const solutions = collection(db, "tasks", taskId, "solutions")
  addDoc(solutions, solution)
}