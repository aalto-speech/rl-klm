from initialParams import initializeParams

def splitUI(batch_number, batch_total, params):
    #params.start = 74.0
    #params.end = 240.0
    if int(batch_total) == 1:
        return int(params.start), int(params.end)

    size_ui_set = (params.end-params.start)/float(batch_total)

    nStart = params.start + int(batch_number)*size_ui_set
    if int(batch_number) == int(batch_total)-1:
        nEnd = params.end
    else: nEnd = nStart + size_ui_set-1

    return int(nStart), int(nEnd)
    
