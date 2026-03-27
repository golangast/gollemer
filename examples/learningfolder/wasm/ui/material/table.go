//go:build js && wasm


package material

import (
	"fmt"
	"syscall/js"
)

type DataTableColumn struct {
	Header string
	Key    string
}

type DataTable struct {
	Columns []DataTableColumn
	Data    []map[string]interface{}
}

func NewDataTable(cols []DataTableColumn, data []map[string]interface{}) *DataTable {
	return &DataTable{Columns: cols, Data: data}
}

func (dt *DataTable) Render() js.Value {
	document := js.Global().Get("document")
	table := document.Call("createElement", "table")
	table.Set("className", "mat-table")

	// Header
	thead := document.Call("createElement", "thead")
	tr := document.Call("createElement", "tr")
	for _, col := range dt.Columns {
		th := document.Call("createElement", "th")
		th.Set("innerText", col.Header)
		tr.Call("appendChild", th)
	}
	thead.Call("appendChild", tr)
	table.Call("appendChild", thead)

	// Body
	tbody := document.Call("createElement", "tbody")
	for _, row := range dt.Data {
		tr := document.Call("createElement", "tr")
		for _, col := range dt.Columns {
			td := document.Call("createElement", "td")
			val := row[col.Key]
			if val != nil {
				td.Set("innerText", fmt.Sprint(val))
			}
			tr.Call("appendChild", td)
		}
		tbody.Call("appendChild", tr)
	}
	table.Call("appendChild", tbody)

	return table
}
