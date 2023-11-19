import { collection, getDocs, query, where } from "firebase/firestore"
import { db } from "../../firebaseConfig"
import { Dataset } from "../../datamodel"

export const fetchDataset = async (name?:string) => {
  if(!name)
    return null
  const datasetDocs = await getDocs(query(collection(db,"datasets"), where("name",'==',name)))
  if(datasetDocs.docs.length){
    const dataset = {...datasetDocs.docs[0].data(), id:datasetDocs.docs[0].id} as Dataset
    return dataset
  }
  else 
    return null  
}