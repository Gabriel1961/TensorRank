import { Component, For } from "solid-js"
import styles from "./TaskPanelDisplay.module.scss"
import TaskTagDisplay from "./TaskTagDisplay"
import { Task, encodeTaskTitleToUrl } from "../../../datamodel"
import { A } from "@solidjs/router"
interface TaskPanelProps {
  task: Task
}

const TaskPanelDisplay: Component<TaskPanelProps> = (props) => {
  return <div class={styles.container}>
    <A href={`/task/${encodeTaskTitleToUrl(props.task.title)}`}>
      <h2 class={styles.h2}>{props.task.title}</h2>
    </A>
    <h3>{props.task.subtitle}</h3>
    <div class={styles.tagsContainer}>
      <For each={props.task.tags}>
        {(tag, _) => <TaskTagDisplay tag={tag} />}
      </For>
    </div>
  </div>
}

export default TaskPanelDisplay;