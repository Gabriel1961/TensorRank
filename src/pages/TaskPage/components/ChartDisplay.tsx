import { Component, For, Show, createSignal } from "solid-js";
import { Dataset } from "../../../datamodel";
import styles from "./ChartDisplay.module.scss";
interface ChartDisplayProps {
  dataset: Dataset
}

export const ChartDisplay: Component<ChartDisplayProps> = (props) => {
  const [getOpen, setOpen] = createSignal<boolean>(false);
  return <div class={styles.container}>
    <button class={styles.button} onClick={() => setOpen(!getOpen())}>View Dataset Charts</button>
    <div class={styles.imageContainer}>
      <Show when={getOpen() && props.dataset}>
        <For each={props.dataset.chartImages}>
          {(src) => <img class={styles.image} src={src} />}
        </For>
      </Show>
    </div>
  </div>
}