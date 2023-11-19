import { Component, Show, createEffect, createSignal } from "solid-js"
import styles from "./DatasetPage.module.scss"
import { useParams } from "@solidjs/router"
import { Dataset, decodeTextTitleToUrl } from "../../datamodel"
import { fetchDataset } from "./DatasetPageFetcher"
import { capitalizeFirstLetter } from "../Datasets/Components/DatasetPanel"

interface DatasetPageParams {
  datasetName?: string 
}

const DatasetPage:Component = () => {
  const params = useParams() as any as DatasetPageParams
  const [getDataset, setDataset] = createSignal<Dataset|null>(null) 
  let longDescription: HTMLDivElement
  
  createEffect(()=>{
    if(!params.datasetName)
      return
    fetchDataset(decodeTextTitleToUrl(params.datasetName)).then(dataset=>{
      if(dataset){
        setDataset( dataset)
        longDescription.innerHTML = dataset.longDescription
      }
    })
  })

  return <div class={styles.container}>
    <Show when={getDataset()}>
      <h2 class={styles.h2}>{capitalizeFirstLetter(getDataset()!.name)} Dataset</h2>
      <p>{getDataset()!.description}</p>
      <div class={styles.description} ref={longDescription!}/>
    </Show>
  </div>
}

export default DatasetPage