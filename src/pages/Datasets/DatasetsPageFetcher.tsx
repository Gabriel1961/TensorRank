import { collection, getDocs, limit, orderBy, query, startAfter } from "firebase/firestore";
import { Dataset } from "../../datamodel";
import { db } from "../../firebaseConfig";

const BATCH_SIZE = 10

export const fetchMoreDatasets = async (currentDatasets: Dataset[]) => {
  const datasets = collection(db, "datasets")
  let datasetsQuery = query(datasets, orderBy("name"), limit(BATCH_SIZE))

  if (currentDatasets.length !== 0) {
    datasetsQuery = query(datasets, startAfter(currentDatasets[currentDatasets.length - 1].name))
  }

  const newDatasets = (await getDocs(datasetsQuery)).docs.map(dataset => ({ ...dataset.data(), id: dataset.id } as Dataset))

  return [...currentDatasets, ...newDatasets]
}