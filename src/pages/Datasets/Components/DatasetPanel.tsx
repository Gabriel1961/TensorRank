import { A } from "@solidjs/router"
import { Component, For } from "solid-js"
import { Dataset, encodeTaskTitleToUrl } from "../../../datamodel"
import TaskTagDisplay from "../../TasksPage/Components/TaskTagDisplay"
import styles from "./DatasetPanel.module.scss"

interface DatasetPanelProps {
  dataset: Dataset
}

export function capitalizeFirstLetter(word:string) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

const DatasetPanel: Component<DatasetPanelProps> = (props) => {
  return <div class={styles.container}>
    <A href={`/dataset/${encodeTaskTitleToUrl(props.dataset.name)}`}>
      <h2>{capitalizeFirstLetter(props.dataset.name)} Dataset</h2>
      <p class={styles.p}>{props.dataset.description}</p>
      <div class={styles.detailsPanels}>
        <p class={styles.roundContainer}>Subject Area: {props.dataset.subject}</p>
        <p class={styles.roundContainer}>Instances: {props.dataset.instances}</p>
        <p class={styles.roundContainer}>Features: {props.dataset.features}</p>
        <p class={styles.roundContainer}>Author: {props.dataset.author}</p>
      </div>
    </A>
    <div class={styles.tagsContainer}>
      <For each={props.dataset.tags}>
        {(tag, _) => <TaskTagDisplay tag={tag} />}
      </For>
    </div>
  </div>
}

export default DatasetPanel;