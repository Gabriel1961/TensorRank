import { collection, getDocs, limit, orderBy, query, where } from "firebase/firestore"
import { db } from "../../firebaseConfig"
import { Solution } from "../../datamodel"


export const fetchScoreboard = async (userId:string|null, taskId?:string) => {
  if(!taskId)
    return []
  const scoreboardCollection = collection(db, "tasks", taskId, "solutions")
  
  let q 
  if(userId){
    q = query(scoreboardCollection, where("author.id","==",userId), orderBy("score","desc"), limit(10))
  }
  else {
    q = query(scoreboardCollection, orderBy("score","desc"), limit(10))
  } 
  
  const solutions = await getDocs(q)
  return solutions.docs.map((sol)=>({...sol.data(), id:sol.id} as Solution)) 
}