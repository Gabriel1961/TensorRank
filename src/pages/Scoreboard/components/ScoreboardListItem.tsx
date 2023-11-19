import { Component, Show, createEffect, createSignal } from "solid-js"
import styles from "./ScoreboardListItem.module.scss"
import { Solution } from "../../../datamodel"
import * as monaco from "monaco-editor"
import { AiFillTrophy } from "solid-icons/ai"

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
      <div class={styles.horizontal2}>
        <Show when={props.idx === 1} >
          <AiFillTrophy size={30} color="#f1c40f" class={styles.trophy} />
        </Show>

        <Show when={props.idx === 2} >
          <AiFillTrophy size={30} color="#95a5a6" class={styles.trophy} />
        </Show>

        <Show when={props.idx === 3} >
          <AiFillTrophy size={30} color="#cd7f32" class={styles.trophy} />
        </Show>
        <h3> {props.idx}. {props.score.author.name}</h3>
      </div>
      <h3>Accuracy: {(100 * props.score.score).toFixed(2)}%</h3>
    </div>
    <Show when={getExpanded()}>
      <div class={styles.editor} ref={editor!}>
      </div>
    </Show>
  </div>
}

export default ScoreboardListItem