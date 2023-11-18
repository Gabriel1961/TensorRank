import { collection, getDocs, limit, orderBy, query, startAfter } from "firebase/firestore";
import { Task } from "../../datamodel";
import { db } from "../../firebaseConfig";

const BATCH_SIZE = 10

export const fetchMoreTasks = async (currentTasks:Task[]) => {
  const datasets = collection(db, "tasks")
  let datasetsQuery = query(datasets, orderBy("title"), limit(BATCH_SIZE))

  if (currentTasks.length !== 0) {
    datasetsQuery = query(datasetsQuery, startAfter(currentTasks[currentTasks.length - 1].title))
  }

  const newTasks = (await getDocs(datasetsQuery)).docs.map(dataset => ({ ...dataset.data(), id: dataset.id } as Task))

  return [...currentTasks, ...newTasks]
}