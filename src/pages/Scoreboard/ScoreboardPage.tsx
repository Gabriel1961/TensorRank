import { useParams } from "@solidjs/router"
import { Component, For, createEffect, createSignal } from "solid-js"
import { useLogin } from "../../contexts/loginContext"
import { Solution } from "../../datamodel"
import { fetchScoreboard } from "./ScoreboardFetcher"
import styles from "./ScoreboardPage.module.scss"
import ScoreboardListItem from "./components/ScoreboardListItem"

interface ScoreBoardPageParams {
  taskId?:string 
}

export const ScoreboardPage: Component = () => {
  const params = useParams() as any as ScoreBoardPageParams
  const [getLoginContext] = useLogin()
  const [getSolutions, setSolutions] = createSignal<Solution[]>()
  const [getPersonalSolutions, setPersonalSolutions] = createSignal<boolean>(false)

  createEffect(()=>{
    fetchScoreboard( getPersonalSolutions() ? getLoginContext().user?.uid ?? null : null, params.taskId).then(scores=> {
      setSolutions(scores)
    })
  })

  return <div class={styles.container} >
    <div style={{margin:"20px", "margin-bottom":"0px", }}>
      <input type="checkbox" id="horns" name="horns" onchange={()=>setPersonalSolutions(!getPersonalSolutions())}/>
      <label style={{"margin-left":"10px"}} for="horns">Only Your Solutions</label>
    </div>
    <For each={getSolutions()}>
      {(solution,idx) => <ScoreboardListItem  idx={idx()+1} score={solution}/>}
    </For>
  </div>
}

export default ScoreboardPage;