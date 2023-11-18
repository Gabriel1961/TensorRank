import { Component, For, Show, createEffect, createSignal } from "solid-js";
import { Dataset } from "../../datamodel";
import DatasetPanel from "./Components/DatasetPanel";
import styles from "./Datasets.module.scss";
import { fetchMoreDatasets } from "./DatasetsPageFetcher";
import Spinner from "../../components/Spinner";

const DatasetsPage: Component = () => {
  const [getDatasets, setDatasets] = createSignal<Dataset[]>([])
  const [getIsLoading, setIsLoading] = createSignal<boolean>(false)

  const loadMoreDatasets = async (datasets: Dataset[]) => {
    setIsLoading(true)
    const newDatasets = await fetchMoreDatasets(datasets)
    setDatasets(newDatasets);
    setIsLoading(false)
  }

  createEffect(() => {
    loadMoreDatasets([])
  })

  return <div class={styles.container}>
    <div class={styles.filterPanel}>
      <h2>Filter</h2>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked/>
        <label style={{ "margin-left": "10px" }} >Classification</label>
      </div>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" />
        <label style={{ "margin-left": "10px" }} >Regression</label>
      </div>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" />
        <label style={{ "margin-left": "10px" }} >Easy</label>
      </div>

      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked/>
        <label style={{ "margin-left": "10px" }} >Medium</label>
      </div>

      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" />
        <label style={{ "margin-left": "10px" }} >Hard</label>
      </div>
    </div>
    <div class={styles.containerList}>
      <For each={getDatasets()}>
        {(dataset) => <DatasetPanel dataset={dataset} />}
      </For>
      <Show when={getIsLoading() === false} fallback={<Spinner />}>
        <button class={styles.loadMore} onClick={() => loadMoreDatasets(getDatasets())}>Load More</button>
      </Show>
    </div>
  </div >
}

export default DatasetsPage