import { Component, Show, createEffect, createSignal } from "solid-js"
import styles from "./ScoreboardListItem.module.scss"
import { Solution } from "../../../datamodel"
import * as monaco from "monaco-editor"

interface ScoreBoardListItemProps {
  score: Solution,
  idx: number
}

const ScoreboardListItem: Component<ScoreBoardListItemProps> = (props) => {
  const [getExpanded, setExpanded] = createSignal<boolean>(false)
  let currentEditor: monaco.editor.IStandaloneCodeEditor | null = null
  let editor: HTMLDivElement

  createEffect(() => {
    if (getExpanded()) {
      monaco.editor.create(editor, { theme: "vs-dark", readOnly: true, value: "\n\n\n" + props.score.sourceCode, language: "javascript" })
    }
    else {
      currentEditor?.dispose()
    }
  })

  return <div class={styles.container} >
    <div class={styles.horizontal} onClick={() => setExpanded(!getExpanded())}> 
      <h3>{props.idx}. {props.score.author.name}</h3>
      <h3>Accuracy: {(100*props.score.score).toFixed(2)}%</h3>
    </div>
    <Show when={getExpanded()}>
      <div class={styles.editor} ref={editor}>
      </div>
    </Show>
  </div>
}

export default ScoreboardListItem