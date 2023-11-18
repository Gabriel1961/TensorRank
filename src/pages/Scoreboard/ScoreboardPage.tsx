import { Component, For, Show, createEffect, createSignal } from "solid-js"
import styles from "./ScoreboardPage.module.scss"
import { useLogin } from "../../contexts/loginContext"
import { Solution } from "../../datamodel"
import ScoreboardListItem from "./components/ScoreboardListItem"
import { fetchScoreboard } from "./ScoreboardFetcher"


export const ScoreboardPage: Component = () => {
  const [getLoginContext] = useLogin()
  const [getSolutions, setSolutions] = createSignal<Solution[]>()
  const [getPersonalSolutions, setPersonalSolutions] = createSignal<boolean>(false)

  createEffect(()=>{
    fetchScoreboard( getPersonalSolutions() ? getLoginContext().user?.uid ?? null : null).then(scores=> {
      setSolutions(scores)
    })
  })

  return <div class={styles.container} >
    <div style={{margin:"20px", "margin-bottom":"0px", }}>
      <input type="checkbox" id="horns" name="horns" onchange={()=>setPersonalSolutions(!getPersonalSolutions())}/>
      <label style={{"margin-left":"10px"}} for="horns">Only Personal Solutions</label>
    </div>
    <For each={getSolutions()}>
      {(solution,idx) => <ScoreboardListItem  idx={idx()+1} score={solution}/>}
    </For>
  </div>
}

export default ScoreboardPage;